"""
Train the models and evaluate them.

    python train.py            # development evaluation (never touches the reserved final-test seeds)
    python train.py --final    # ONE-TIME final test on the reserved seeds (>= 900000). Do this last.

The model's job: given a workload's features, RANK the candidate schedulers.
We keep the top-k as a shortlist, simulate only those k, and pick the best of them.

Rankers compared:
  rf_classifier : random forest that predicts the label; ranks by predicted probability
  rf_regressor  : random forest that predicts every candidate's score; ranks lowest-first
  tree          : single decision tree classifier (easy to read, see tree_rules.txt)
  static        : ALWAYS the same shortlist = candidates with the best average score on the
                  training data. If the ML cannot beat this, the ML adds nothing.
  random        : random shortlist

Metrics (all on workloads the model never trained on):
  hit rate  = how often the shortlist contains a truly best scheduler
  regret    = (score of best scheduler in shortlist) - (score of the true best); 0 is perfect
  raw wt/rt = actual average waiting / response time achieved (lower is better)
"""
import argparse
import numpy as np
import pandas as pd
import joblib
import datetime
import os
import platform
import sklearn
from sklearn.tree import export_text
from sklearn.model_selection import KFold

from config import FINAL_TEST_SEED_START, FAMILIES
from candidates import LABEL_NAMES as CANDS
from features import FEATURE_COLS
from scoring import add_labels
from modeling import fit_rankers, class_proba_matrix, SCORE_COLS

KS = [1, 2, 3, 5]


# ---- rankers: each returns a matrix (n_rows, n_candidates); LOWER = predicted better ----
def rank_matrices(models, test, rng):
    X = test[FEATURE_COLS]
    return {
        "rf_classifier": -class_proba_matrix(models["clf"], X),
        "rf_regressor": models["reg"].predict(X),
        "tree": -class_proba_matrix(models["tree"], X),
        "static": np.tile(models["static"], (len(test), 1)),
        "random": rng.random((len(test), len(CANDS))),
    }


# ---- evaluation ----
def evaluate(rank, df):
    S = df[SCORE_COLS].to_numpy()
    WT = df[[f"{c}__avg_wt" for c in CANDS]].to_numpy()
    RT = df[[f"{c}__avg_rt" for c in CANDS]].to_numpy()
    order = np.argsort(rank, axis=1, kind="stable")
    best = S.min(axis=1)
    rows = {}
    for k in KS:
        top = order[:, :k]
        s_top = np.take_along_axis(S, top, axis=1)
        pick = top[np.arange(len(df)), s_top.argmin(axis=1)]          # best of the shortlist
        idx = np.arange(len(df))
        rows[k] = {
            "hit": float((s_top.min(axis=1) <= best + 1e-12).mean()),
            "regret": float((s_top.min(axis=1) - best).mean()),
            "wt": float(WT[idx, pick].mean()),
            "rt": float(RT[idx, pick].mean()),
        }
    return rows


def oracle_row(df):
    S = df[SCORE_COLS].to_numpy()
    pick = S.argmin(axis=1)
    idx = np.arange(len(df))
    WT = df[[f"{c}__avg_wt" for c in CANDS]].to_numpy()
    RT = df[[f"{c}__avg_rt" for c in CANDS]].to_numpy()
    return {"wt": float(WT[idx, pick].mean()), "rt": float(RT[idx, pick].mean())}


def report(title, all_ranks, df):
    print(f"\n=== {title} ({len(df)} workloads, {len(CANDS)} candidate schedulers) ===")
    print(f"{'ranker':15s} {'k':>2s} {'hit rate':>9s} {'regret':>8s} {'avg WT':>8s} {'avg RT':>8s}  sims run")
    for name, rank in all_ranks.items():
        res = evaluate(rank, df)
        for k in KS:
            r = res[k]
            print(f"{name:15s} {k:2d} {r['hit'] * 100:8.1f}% {r['regret']:8.4f} {r['wt']:8.2f} {r['rt']:8.2f}  {k}/{len(CANDS)}")
        print()
    o = oracle_row(df)
    print(f"{'oracle':15s} {'-':>2s} {'100.0%':>9s} {0:8.4f} {o['wt']:8.2f} {o['rt']:8.2f}  {len(CANDS)}/{len(CANDS)} (runs everything)")
    if "SRTF__avg_wt" in df:
        print(f"{'SRTF reference':15s} (needs true burst times; real OS cannot do this) "
              f"avg WT {df['SRTF__avg_wt'].mean():.2f}, avg RT {df['SRTF__avg_rt'].mean():.2f}")


def cross_validate(df, folds, rng):
    """folds: list of (train_index, test_index). Predictions are pooled over all folds."""
    pooled = {k: np.zeros((len(df), len(CANDS))) for k in
              ["rf_classifier", "rf_regressor", "tree", "static", "random"]}
    for tr, te in folds:
        models = fit_rankers(df.iloc[tr])
        ranks = rank_matrices(models, df.iloc[te], rng)
        for k, m in ranks.items():
            pooled[k][te] = m
    return pooled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true", help="evaluate on the reserved final-test seeds (do once!)")
    args = ap.parse_args()

    full = add_labels(pd.read_csv("data/dataset.csv"))
    dev = full[full.seed < FINAL_TEST_SEED_START].reset_index(drop=True)
    final = full[full.seed >= FINAL_TEST_SEED_START].reset_index(drop=True)
    rng = np.random.default_rng(0)
    print(f"development workloads: {len(dev)}   reserved final-test workloads: {len(final)}")

    if not args.final:
        # 1) random 5-fold: did it learn anything?
        folds = list(KFold(5, shuffle=True, random_state=0).split(dev))
        report("5-fold cross-validation (random split)", cross_validate(dev, folds, rng), dev)

        # 2) leave-one-family-out: does it work on a KIND of workload it never saw?
        fam_folds = []
        for f in FAMILIES:
            te = np.where(dev.family == f)[0]
            tr = np.where(dev.family != f)[0]
            fam_folds.append((tr, te))
        report("leave-one-family-out (tough test)", cross_validate(dev, fam_folds, rng), dev)

        # fit on everything, save, and show what the model learned
        models = fit_rankers(dev)
        os.makedirs("models", exist_ok=True)
        meta = {"sklearn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__,
                "python": platform.python_version(), "n_dev_workloads": int(len(dev)),
                "created": datetime.date.today().isoformat(), "demo": False}
        joblib.dump({"models": models, "cands": CANDS, "features": FEATURE_COLS, "meta": meta},
                    "models/ranker.joblib", compress=3)
        with open("models/tree_rules.txt", "w") as fh:
            fh.write(export_text(models["tree"], feature_names=FEATURE_COLS))
        print("\nFeature importance (random forest classifier):")
        imp = sorted(zip(FEATURE_COLS, models["clf"].feature_importances_), key=lambda t: -t[1])
        for f, v in imp:
            print(f"  {f:18s} {v:.3f}")
        print("\nSaved models/ranker.joblib and models/tree_rules.txt")
    else:
        if len(final) == 0:
            raise SystemExit("No rows with seed >= 900000 in data/dataset.csv")
        models = fit_rankers(dev)
        report("FINAL TEST (trained on all development data)", rank_matrices(models, final, rng), final)


if __name__ == "__main__":
    main()
